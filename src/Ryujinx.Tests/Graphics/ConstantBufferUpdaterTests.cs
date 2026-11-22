using NUnit.Framework;
using Ryujinx.Common.Configuration;
using Ryujinx.Graphics.GAL;
using Ryujinx.Graphics.Gpu;
using Ryujinx.Graphics.Gpu.Engine;
using Ryujinx.Graphics.Gpu.Engine.Threed;
using Ryujinx.Graphics.Gpu.Engine.Types;
using Ryujinx.Graphics.Gpu.Memory;
using Ryujinx.Memory;
using System;
using System.Runtime.InteropServices;
using CpuMemoryManager = Ryujinx.Cpu.Jit.MemoryManager;
using GpuMemoryManager = Ryujinx.Graphics.Gpu.Memory.MemoryManager;

namespace Ryujinx.Tests.Graphics
{
    [TestFixture]
    public class ConstantBufferUpdaterTests
    {
        private const ulong Pid = 1;
        private const ulong AddressSpaceSize = 0x100000;
        private const ulong BackingMemorySize = 0x10000;

        private MemoryBlock _backingMemory;
        private CpuMemoryManager _cpuMemory;
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

            _gpuContext = new GpuContext(new StubRenderer(), default);
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

        private int[] ReadCpuInts(ulong cpuAddress, int count)
        {
            int[] result = new int[count];
            _cpuMemory.Read(cpuAddress, MemoryMarshal.Cast<int, byte>(result.AsSpan()));
            return result;
        }

        [Test]
        public void Update_SpanLargerThanCache_WritesAllDataAndAdvancesOffset()
        {
            const ulong gpuVa = 0x10000;
            const ulong cpuPa = 0x2000;
            const int wordCount = 1024; // Larger than 512-int staging cache

            _gpuMemory.Map(cpuPa, gpuVa, 0x4000, PteKind.Pitch);
            SetUniformBuffer(gpuVa, wordCount * 4);

            int[] expected = new int[wordCount];
            for (int i = 0; i < wordCount; i++)
            {
                expected[i] = i + 1;
            }

            Assert.DoesNotThrow(() => _updater.Update(expected));
            _updater.FlushUboDirty();

            Assert.That(_state.State.UniformBufferState.Offset, Is.EqualTo(wordCount * 4));
            Assert.That(ReadCpuInts(cpuPa, wordCount), Is.EqualTo(expected));
        }

        [Test]
        public void Update_BufferedDataFollowedByOversizedSpan_FlushesPriorDataAndWritesAll()
        {
            const ulong gpuVa = 0x10000;
            const ulong cpuPa = 0x2000;
            const int prefixCount = 10;
            const int largeCount = 600;
            const int suffixCount = 5;
            const int totalCount = prefixCount + largeCount + suffixCount;

            _gpuMemory.Map(cpuPa, gpuVa, 0x4000, PteKind.Pitch);
            SetUniformBuffer(gpuVa, totalCount * 4);

            int[] expected = new int[totalCount];
            for (int i = 0; i < totalCount; i++)
            {
                expected[i] = (i + 1) * 7;
            }

            _updater.Update(expected.AsSpan(0, prefixCount));
            _updater.Update(expected.AsSpan(prefixCount, largeCount));
            _updater.Update(expected.AsSpan(prefixCount + largeCount, suffixCount));
            _updater.FlushUboDirty();

            Assert.That(_state.State.UniformBufferState.Offset, Is.EqualTo(totalCount * 4));
            Assert.That(ReadCpuInts(cpuPa, totalCount), Is.EqualTo(expected));
        }

        [Test]
        public void Update_OversizedSpanAcrossNonContiguousPages_WritesEachMappedPage()
        {
            const ulong gpuVa = 0x20000;
            const ulong firstPagePa = 0x4000;
            const ulong secondPagePa = 0x0000;
            const int pageWords = (int)(GpuMemoryManager.PageSize / 4);
            const int totalWords = 1500; // Crosses 4 KB page boundary and exceeds 512-int cache

            _gpuMemory.Map(firstPagePa, gpuVa, GpuMemoryManager.PageSize, PteKind.Pitch);
            _gpuMemory.Map(secondPagePa, gpuVa + GpuMemoryManager.PageSize, GpuMemoryManager.PageSize, PteKind.Pitch);
            SetUniformBuffer(gpuVa, totalWords * 4);

            int[] expected = new int[totalWords];
            for (int i = 0; i < totalWords; i++)
            {
                expected[i] = 0x10000 + i;
            }

            Assert.DoesNotThrow(() => _updater.Update(expected));
            _updater.FlushUboDirty();

            Assert.That(_state.State.UniformBufferState.Offset, Is.EqualTo(totalWords * 4));
            Assert.That(ReadCpuInts(firstPagePa, pageWords), Is.EqualTo(expected[..pageWords]));
            Assert.That(ReadCpuInts(secondPagePa, totalWords - pageWords), Is.EqualTo(expected[pageWords..]));
        }

        [Test]
        public void Update_EmptySpan_DoesNotModifyOffsetOrThrow()
        {
            const ulong gpuVa = 0x10000;
            const ulong cpuPa = 0x2000;

            _gpuMemory.Map(cpuPa, gpuVa, GpuMemoryManager.PageSize, PteKind.Pitch);
            SetUniformBuffer(gpuVa, 0x1000, offset: 16);

            Assert.DoesNotThrow(() => _updater.Update(ReadOnlySpan<int>.Empty));
            _updater.FlushUboDirty();

            Assert.That(_state.State.UniformBufferState.Offset, Is.EqualTo(16));
        }

        private sealed class StubRenderer : IRenderer
        {
            private ulong _nextBufferHandle = 1;

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
            public void SetBufferData(BufferHandle buffer, int offset, ReadOnlySpan<byte> data) { }
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
