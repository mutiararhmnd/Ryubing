using NUnit.Framework;
using Ryujinx.Common.Configuration;
using Ryujinx.Graphics.GAL;
using Ryujinx.Graphics.Gpu;
using Ryujinx.Graphics.Gpu.Engine;
using Ryujinx.Graphics.Gpu.Engine.GPFifo;
using Ryujinx.Graphics.Gpu.Engine.Threed;
using Ryujinx.Graphics.Gpu.Engine.Types;
using Ryujinx.Graphics.Gpu.Memory;
using Ryujinx.Memory;
using Ryujinx.Memory.Range;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using CpuMemoryManager = Ryujinx.Cpu.Jit.MemoryManager;
using GpuMemoryManager = Ryujinx.Graphics.Gpu.Memory.MemoryManager;

namespace Ryujinx.Tests.Graphics
{
    [TestFixture]
    public class ConstantBufferUpdaterQaTests
    {
        private const ulong Pid = 1;
        private const ulong AddressSpaceSize = 0x100000;
        private const ulong BackingMemorySize = 0x20000;

        private const int UniformBufferStateMethodOffset = 0x8e0;
        private const int UniformBufferUpdateDataMethodOffset = 0x8e4;

        private MemoryBlock _backingMemory;
        private CpuMemoryManager _cpuMemory;
        private RecordingRenderer _renderer;
        private GpuContext _gpuContext;
        private GpuMemoryManager _gpuMemory;
        private GpuChannel _channel;
        private DeviceStateWithShadow<ThreedClassState> _state;
        private ConstantBufferUpdater _updater;

        [SetUp]
        public void Setup()
        {
            _backingMemory = new MemoryBlock(BackingMemorySize);
            _cpuMemory = new CpuMemoryManager(_backingMemory, AddressSpaceSize);
            _cpuMemory.Map(0, 0, BackingMemorySize, MemoryMapFlags.None);

            _renderer = new RecordingRenderer();
            _gpuContext = new GpuContext(_renderer, default);
            _gpuContext.RegisterProcess(Pid, _cpuMemory);

            _gpuMemory = _gpuContext.CreateMemoryManager(Pid, AddressSpaceSize);
            _channel = _gpuContext.CreateChannel();
            _channel.BindMemory(_gpuMemory);

            _state = new DeviceStateWithShadow<ThreedClassState>();
            _updater = new ConstantBufferUpdater(_channel, _state);
        }

        [TearDown]
        public void TearDown()
        {
            _channel.Dispose();
            _gpuContext.UnregisterProcess(Pid);
            _gpuContext.Dispose();
            _backingMemory.Dispose();
        }

        private void SetUniformBuffer(ulong gpuVa, int size, int offset = 0)
        {
            _state.State.UniformBufferState = new UniformBufferState
            {
                Size = size,
                Address = new GpuVa
                {
                    Low = (uint)gpuVa,
                    High = (uint)(gpuVa >> 32),
                },
                Offset = offset,
            };
        }

        private void SyncHostBuffer(ulong gpuVa, ulong size)
        {
            MultiRange range = _gpuMemory.Physical.BufferCache.TranslateAndCreateBuffer(_gpuMemory, gpuVa, size, BufferStage.Vertex);
            _gpuMemory.Physical.BufferCache.GetBufferRange(range, BufferStage.Vertex);
        }

        private int[] ReadCpuInts(ulong cpuAddress, int count)
        {
            int[] result = new int[count];
            _cpuMemory.Read(cpuAddress, MemoryMarshal.Cast<int, byte>(result.AsSpan()));
            return result;
        }

        private static int PackMethod(SecOp secOp, int subChannel, int methodAddress, int count)
        {
            uint value = ((uint)secOp << 29) |
                         (((uint)count & 0x1FFF) << 16) |
                         (((uint)subChannel & 0x7) << 13) |
                         ((uint)methodAddress & 0xFFF);
            return unchecked((int)value);
        }

        private static int[] MakePattern(int count, int seed = 1)
        {
            int[] data = new int[count];
            for (int i = 0; i < count; i++)
            {
                data[i] = unchecked((i + 1) * 31 + seed);
            }
            return data;
        }

        // --- 1. Unmapped (PteUnmapped) pages ---

        [Test]
        public void WriteDirect_MappedUnmappedMappedPages_WritesMappedPagesAndSkipsUnmapped()
        {
            const ulong gpuVa = 0x30000;
            const ulong page0Pa = 0x2000;
            const ulong page2Pa = 0x6000;
            const int pageWords = (int)(GpuMemoryManager.PageSize / 4); // 1024 ints
            const int totalWords = pageWords * 3; // 3072 ints across 3 pages; middle page unmapped

            _gpuMemory.Map(page0Pa, gpuVa, GpuMemoryManager.PageSize, PteKind.Pitch);
            // Page 1 (gpuVa + PageSize) intentionally left unmapped (PteUnmapped)
            _gpuMemory.Map(page2Pa, gpuVa + GpuMemoryManager.PageSize * 2, GpuMemoryManager.PageSize, PteKind.Pitch);

            // Pre-create buffers for mapped pages so we can also verify ForceDirty on them
            SyncHostBuffer(gpuVa, GpuMemoryManager.PageSize);
            SyncHostBuffer(gpuVa + GpuMemoryManager.PageSize * 2, GpuMemoryManager.PageSize);
            _renderer.SetBufferDataCalls.Clear();

            SetUniformBuffer(gpuVa, totalWords * 4);
            int[] expected = MakePattern(totalWords, seed: 100);

            Assert.DoesNotThrow(() => _updater.Update(expected));
            _updater.FlushUboDirty();

            Assert.That(_state.State.UniformBufferState.Offset, Is.EqualTo(totalWords * 4));
            Assert.That(ReadCpuInts(page0Pa, pageWords), Is.EqualTo(expected[..pageWords]));
            Assert.That(ReadCpuInts(page2Pa, pageWords), Is.EqualTo(expected[(pageWords * 2)..]));

            // Syncing mapped pages should upload updated data via SetBufferData
            SyncHostBuffer(gpuVa, GpuMemoryManager.PageSize);
            SyncHostBuffer(gpuVa + GpuMemoryManager.PageSize * 2, GpuMemoryManager.PageSize);

            Assert.That(_renderer.SetBufferDataCalls.Count, Is.EqualTo(2));
            Assert.That(MemoryMarshal.Cast<byte, int>(_renderer.SetBufferDataCalls[0].Data).ToArray(), Is.EqualTo(expected[..pageWords]));
            Assert.That(MemoryMarshal.Cast<byte, int>(_renderer.SetBufferDataCalls[1].Data).ToArray(), Is.EqualTo(expected[(pageWords * 2)..]));
        }

        [Test]
        public void WriteDirect_FullyUnmappedPages_DoesNotThrowAndAdvancesOffset()
        {
            const ulong gpuVa = 0x50000; // Not mapped
            const int wordCount = 1024;

            SetUniformBuffer(gpuVa, wordCount * 4);
            int[] payload = MakePattern(wordCount, seed: 200);
            _renderer.SetBufferDataCalls.Clear();

            Assert.DoesNotThrow(() => _updater.Update(payload));
            _updater.FlushUboDirty();

            Assert.That(_state.State.UniformBufferState.Offset, Is.EqualTo(wordCount * 4));
            Assert.That(_renderer.SetBufferDataCalls, Is.Empty);
        }

        [Test]
        public void WriteDirect_StartsUnmappedEndsMapped_WritesMappedTail()
        {
            const ulong gpuVa = 0x60000;
            const ulong page1Pa = 0x8000;
            const int pageWords = (int)(GpuMemoryManager.PageSize / 4);
            const int totalWords = pageWords * 2;

            // Page 0 unmapped, Page 1 mapped
            _gpuMemory.Map(page1Pa, gpuVa + GpuMemoryManager.PageSize, GpuMemoryManager.PageSize, PteKind.Pitch);
            SetUniformBuffer(gpuVa, totalWords * 4);
            int[] expected = MakePattern(totalWords, seed: 300);

            Assert.DoesNotThrow(() => _updater.Update(expected));
            _updater.FlushUboDirty();

            Assert.That(_state.State.UniformBufferState.Offset, Is.EqualTo(totalWords * 4));
            Assert.That(ReadCpuInts(page1Pa, pageWords), Is.EqualTo(expected[pageWords..]));
        }

        // --- 2. BufferCache.ForceDirty & Redundancy Check ---

        [Test]
        public void WriteDirect_ForcesDirtyOnCachedBuffer_AndSkipsRedundantWrite()
        {
            const ulong gpuVa = 0x10000;
            const ulong cpuPa = 0x2000;
            const int wordCount = 1024;
            const ulong byteSize = wordCount * 4;

            _gpuMemory.Map(cpuPa, gpuVa, 0x4000, PteKind.Pitch);

            // Create and synchronize host buffer initially (all zeros)
            SyncHostBuffer(gpuVa, byteSize);
            _renderer.SetBufferDataCalls.Clear();

            SetUniformBuffer(gpuVa, (int)byteSize);
            int[] expected = MakePattern(wordCount, seed: 400);

            _updater.Update(expected);

            // Trigger buffer sync to verify ForceDirty caused host upload
            SyncHostBuffer(gpuVa, byteSize);
            Assert.That(_renderer.SetBufferDataCalls.Count, Is.EqualTo(1));
            Assert.That(_renderer.SetBufferDataCalls[0].Offset, Is.EqualTo(0));
            Assert.That(MemoryMarshal.Cast<byte, int>(_renderer.SetBufferDataCalls[0].Data).ToArray(), Is.EqualTo(expected));

            // Second identical write should be skipped by WriteWithRedundancyCheck -> no ForceDirty -> no SetBufferData
            _renderer.SetBufferDataCalls.Clear();
            SetUniformBuffer(gpuVa, (int)byteSize, offset: 0);
            _updater.Update(expected);
            SyncHostBuffer(gpuVa, byteSize);
            Assert.That(_renderer.SetBufferDataCalls, Is.Empty);

            // Mutate a single word and rewrite -> must ForceDirty and upload
            expected[wordCount - 1] ^= 0x55AA55AA;
            SetUniformBuffer(gpuVa, (int)byteSize, offset: 0);
            _updater.Update(expected);
            SyncHostBuffer(gpuVa, byteSize);
            Assert.That(_renderer.SetBufferDataCalls.Count, Is.EqualTo(1));
            Assert.That(MemoryMarshal.Cast<byte, int>(_renderer.SetBufferDataCalls[0].Data).ToArray(), Is.EqualTo(expected));
        }

        [Test]
        public void WriteDirect_NonContiguousPages_ForcesDirtyPerSubRange()
        {
            const ulong gpuVa = 0x20000;
            const ulong firstPagePa = 0x4000;
            const ulong secondPagePa = 0x0000;
            const int pageWords = (int)(GpuMemoryManager.PageSize / 4);
            const int totalWords = 1500;
            const int secondPageWords = totalWords - pageWords;

            _gpuMemory.Map(firstPagePa, gpuVa, GpuMemoryManager.PageSize, PteKind.Pitch);
            _gpuMemory.Map(secondPagePa, gpuVa + GpuMemoryManager.PageSize, GpuMemoryManager.PageSize, PteKind.Pitch);

            SyncHostBuffer(gpuVa, GpuMemoryManager.PageSize);
            SyncHostBuffer(gpuVa + GpuMemoryManager.PageSize, (ulong)secondPageWords * 4);
            _renderer.SetBufferDataCalls.Clear();

            SetUniformBuffer(gpuVa, totalWords * 4);
            int[] expected = MakePattern(totalWords, seed: 500);

            _updater.Update(expected);

            SyncHostBuffer(gpuVa, GpuMemoryManager.PageSize);
            SyncHostBuffer(gpuVa + GpuMemoryManager.PageSize, (ulong)secondPageWords * 4);

            Assert.That(_renderer.SetBufferDataCalls.Count, Is.EqualTo(2));
            Assert.That(MemoryMarshal.Cast<byte, int>(_renderer.SetBufferDataCalls[0].Data).ToArray(), Is.EqualTo(expected[..pageWords]));
            Assert.That(MemoryMarshal.Cast<byte, int>(_renderer.SetBufferDataCalls[1].Data).ToArray(), Is.EqualTo(expected[pageWords..]));
        }

        // --- 3. GPFifoProcessor.Process fast path with > 512 ints ---

        [TestCase(511)]
        [TestCase(512)]
        [TestCase(513)]
        [TestCase(1024)]
        [TestCase(4096)]
        public void GPFifoProcess_FastPathUboUpdate_WritesAllDataAndFollowUpCommand(int largeCount)
        {
            const ulong gpuVa = 0x10000;
            const ulong cpuPa = 0x2000;
            const int followUpCount = 8;
            int totalWords = largeCount + followUpCount;

            _gpuMemory.Map(cpuPa, gpuVa, 0x8000, PteKind.Pitch);

            int[] expected = MakePattern(totalWords, seed: largeCount);

            var commands = new List<int>
            {
                // IncMethod at 0x8e0 (UniformBufferState): Size, Address.High, Address.Low, Offset
                PackMethod(SecOp.IncMethod, 0, UniformBufferStateMethodOffset, 4),
                totalWords * 4,
                unchecked((int)(gpuVa >> 32)),
                unchecked((int)(gpuVa & 0xFFFFFFFF)),
                0,

                // NonIncMethod at 0x8e4 (UniformBufferUpdateData) -> hits TryFastUniformBufferUpdate
                PackMethod(SecOp.NonIncMethod, 0, UniformBufferUpdateDataMethodOffset, largeCount),
            };
            commands.AddRange(expected[..largeCount]);

            // Follow-up fast UBO update in the same command buffer to confirm index advance and continuation
            commands.Add(PackMethod(SecOp.NonIncMethod, 0, UniformBufferUpdateDataMethodOffset, followUpCount));
            commands.AddRange(expected[largeCount..]);

            using var processor = new GPFifoProcessor(_gpuContext, _channel);
            Assert.DoesNotThrow(() => processor.Process(0x80000, CollectionsMarshal.AsSpan(commands)));

            Assert.That(ReadCpuInts(cpuPa, totalWords), Is.EqualTo(expected));
        }

        [Test]
        public void GPFifoProcess_FastPathBufferedThenOversizedThenBuffered_WritesAllInOrder()
        {
            const ulong gpuVa = 0x10000;
            const ulong cpuPa = 0x2000;
            const int prefixCount = 16;
            const int largeCount = 700;
            const int suffixCount = 12;
            const int totalWords = prefixCount + largeCount + suffixCount;

            _gpuMemory.Map(cpuPa, gpuVa, 0x4000, PteKind.Pitch);
            int[] expected = MakePattern(totalWords, seed: 777);

            var commands = new List<int>
            {
                PackMethod(SecOp.IncMethod, 0, UniformBufferStateMethodOffset, 4),
                totalWords * 4,
                unchecked((int)(gpuVa >> 32)),
                unchecked((int)(gpuVa & 0xFFFFFFFF)),
                0,
                PackMethod(SecOp.NonIncMethod, 0, UniformBufferUpdateDataMethodOffset, prefixCount),
            };
            commands.AddRange(expected[..prefixCount]);
            commands.Add(PackMethod(SecOp.NonIncMethod, 0, UniformBufferUpdateDataMethodOffset, largeCount));
            commands.AddRange(expected[prefixCount..(prefixCount + largeCount)]);
            commands.Add(PackMethod(SecOp.NonIncMethod, 0, UniformBufferUpdateDataMethodOffset, suffixCount));
            commands.AddRange(expected[(prefixCount + largeCount)..]);

            using var processor = new GPFifoProcessor(_gpuContext, _channel);
            Assert.DoesNotThrow(() => processor.Process(0x80000, CollectionsMarshal.AsSpan(commands)));

            Assert.That(ReadCpuInts(cpuPa, totalWords), Is.EqualTo(expected));
        }

        // --- 4. Boundary and unaligned offset cases ---

        [Test]
        public void Update_UnalignedOffsetStraddlingNonContiguousPages_WritesCorrectSlices()
        {
            const ulong gpuVa = 0x40000;
            const ulong page0Pa = 0xA000;
            const ulong page1Pa = 0x4000;
            const int initialOffsetWords = 700; // Starts past halfway through page 0
            const int writeWords = 600;         // > 512 ints; 324 words in page 0, 276 words in page 1
            const int pageWords = (int)(GpuMemoryManager.PageSize / 4);
            const int wordsInPage0 = pageWords - initialOffsetWords; // 324
            const int wordsInPage1 = writeWords - wordsInPage0;      // 276

            _gpuMemory.Map(page0Pa, gpuVa, GpuMemoryManager.PageSize, PteKind.Pitch);
            _gpuMemory.Map(page1Pa, gpuVa + GpuMemoryManager.PageSize, GpuMemoryManager.PageSize, PteKind.Pitch);
            SetUniformBuffer(gpuVa, (int)(GpuMemoryManager.PageSize * 2), offset: initialOffsetWords * 4);

            int[] expected = MakePattern(writeWords, seed: 999);

            Assert.DoesNotThrow(() => _updater.Update(expected));
            _updater.FlushUboDirty();

            Assert.That(_state.State.UniformBufferState.Offset, Is.EqualTo((initialOffsetWords + writeWords) * 4));
            Assert.That(ReadCpuInts(page0Pa + (ulong)(initialOffsetWords * 4), wordsInPage0), Is.EqualTo(expected[..wordsInPage0]));
            Assert.That(ReadCpuInts(page1Pa, wordsInPage1), Is.EqualTo(expected[wordsInPage0..]));
        }

        private sealed class RecordingRenderer : IRenderer
        {
            private ulong _nextBufferHandle = 1;

            public List<(BufferHandle Buffer, int Offset, byte[] Data)> SetBufferDataCalls { get; } = new();

            public event EventHandler<ScreenCaptureImageInfo> ScreenCaptured { add { } remove { } }
            public bool PreferThreading => false;
            public IPipeline Pipeline => null;
            public IWindow Window => null;
            public uint ProgramCount => 0;

            public void BackgroundContextAction(Action action, bool alwaysBackground = false) => action();
            public BufferHandle CreateBuffer(int size, BufferAccess access = BufferAccess.Default) => UnsafeHandle(_nextBufferHandle++);
            public BufferHandle CreateBuffer(nint pointer, int size) => UnsafeHandle(_nextBufferHandle++);
            public BufferHandle CreateBufferSparse(ReadOnlySpan<BufferRange> storageBuffers) => UnsafeHandle(_nextBufferHandle++);
            public IImageArray CreateImageArray(int size, bool isBuffer) => null;
            public IProgram CreateProgram(ShaderSource[] shaders, ShaderInfo info) => null;
            public ISampler CreateSampler(SamplerCreateInfo info) => null;
            public ITexture CreateTexture(TextureCreateInfo info) => null;
            public ITextureArray CreateTextureArray(int size, bool isBuffer) => null;
            public bool PrepareHostMapping(nint address, ulong size) => false;
            public void CreateSync(ulong id, bool strict) { }
            public void DeleteBuffer(BufferHandle buffer) { }
            public PinnedSpan<byte> GetBufferData(BufferHandle buffer, int offset, int size) => default;
            public Capabilities GetCapabilities() => default;
            public ulong GetCurrentSync() => 0;
            public HardwareInfo GetHardwareInfo() => default;
            public IProgram LoadProgramBinary(byte[] programBinary, bool hasFragmentShader, ShaderInfo info) => null;
            public void SetBufferData(BufferHandle buffer, int offset, ReadOnlySpan<byte> data)
            {
                SetBufferDataCalls.Add((buffer, offset, data.ToArray()));
            }
            public void UpdateCounters() { }
            public void PreFrame() { }
            public ICounterEvent ReportCounter(CounterType type, EventHandler<ulong> resultHandler, float divisor, bool hostReserved) => null;
            public void ResetCounter(CounterType type) { }
            public void WaitSync(ulong id) { }
            public void Initialize(GraphicsDebugLevel logLevel) { }
            public void SetInterruptAction(Action<Action> interruptAction) { }
            public void Screenshot() { }
            public void Dispose() { }

            private static BufferHandle UnsafeHandle(ulong value)
            {
                return System.Runtime.CompilerServices.Unsafe.As<ulong, BufferHandle>(ref value);
            }
        }
    }
}
